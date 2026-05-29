/**
 * Deny-by-default auth gate.
 *
 * When DASHBOARD_AUTH_ENABLED=true, EVERY route requires the session cookie
 * EXCEPT an explicit public allowlist:
 *   - "/"                  (Overview — public landing)
 *   - "/results"           (public clean-result catalog)
 *   - "/results/*"         (public clean-result detail)
 *   - "/sign-in"           (login page)
 *   - "/api/auth/password" (login endpoint — must be reachable to sign in)
 *   - Next assets          (/_next/*, /favicon*, static files)
 *
 * Everything else — /tasks, /docs, /updates, /literature, /preview, the
 * task edit/plan routes, and ALL other /api/* (incl. /api/sidecar/*,
 * /api/chat-token, /api/docs/*, /api/log/*, /api/updates/*) — is gated.
 *
 * FAILS CLOSED:
 *   - A path that is neither static-asset nor on the public allowlist is
 *     gated by default. Adding a new private route requires no proxy change;
 *     forgetting one fails closed (gated), not open.
 *   - When auth is on but AUTH_SECRET is missing/short, redirect to a
 *     misconfig sign-in (do not silently pass).
 *
 * Next 16 runs proxy on the nodejs runtime (per upgrade docs), so jose's full
 * crypto API is available here just like in the API routes that import
 * lib/auth.
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

/**
 * Public allowlist. A request is public iff its pathname matches one of these
 * exact paths or prefixes. Anything else is gated.
 *
 * Static assets (`/_next/*`, `/favicon*`, files with an extension) are already
 * excluded by the matcher below, but we re-check `_next`/`favicon` here as a
 * belt-and-suspenders guard in case the matcher is ever loosened.
 */
function isPublicPath(pathname: string): boolean {
  // Overview (exact root only).
  if (pathname === "/") return true;
  // Public clean-result catalog + detail.
  if (pathname === "/results" || pathname.startsWith("/results/")) return true;
  // Login page + login endpoint (must be reachable to authenticate).
  if (pathname === "/sign-in" || pathname.startsWith("/sign-in/")) return true;
  if (pathname === "/api/auth/password") return true;
  // Static assets (defensive; normally excluded by the matcher).
  if (pathname.startsWith("/_next/")) return true;
  if (pathname === "/favicon.ico" || pathname.startsWith("/favicon")) return true;
  return false;
}

export async function proxy(request: NextRequest) {
  if (!authEnabled()) return NextResponse.next();

  const { pathname } = request.nextUrl;

  // Public allowlist: skip the auth check entirely.
  if (isPublicPath(pathname)) return NextResponse.next();

  const secret = authSecretBytes();
  if (!secret) {
    // Misconfigured: auth is on but no secret. Fail closed.
    if (pathname.startsWith("/api/")) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
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
  // Run on every path EXCEPT Next's internal static assets + metadata files.
  // The in-function `isPublicPath` allowlist then decides which of the matched
  // paths are public; everything else is gated (deny-by-default). API routes
  // are intentionally MATCHED here (not excluded) so the function can gate
  // them. `_next/data` runs the proxy regardless of this pattern (per Next
  // docs), which is the safe direction for a deny-by-default gate.
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|favicon.png|sitemap.xml|robots.txt).*)",
  ],
};
