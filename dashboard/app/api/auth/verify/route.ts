/**
 * GET /api/auth/verify?token=<jwt>
 *
 * Verifies a magic-link token, sets the session cookie, redirects to
 * `/updates`. On failure, redirects to `/sign-in?error=<reason>`.
 */
import { NextResponse } from "next/server";
import { verifyMagicLinkAndSetSession } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const token = url.searchParams.get("token") ?? "";
  if (!token) {
    return NextResponse.redirect(new URL("/sign-in?error=missing_token", url.origin));
  }
  const result = await verifyMagicLinkAndSetSession(token);
  if (!result.ok) {
    return NextResponse.redirect(
      new URL(`/sign-in?error=${encodeURIComponent(result.reason)}`, url.origin),
    );
  }
  return NextResponse.redirect(new URL("/updates", url.origin));
}
