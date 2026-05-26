/**
 * Shared-password sign-in. POST {password} → on match, set session cookie
 * and return {ok:true}. On mismatch, 401. Alternative to magic link for
 * users who haven't configured Resend.
 */
import { verifyPasswordAndSetSession } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  let body: { password?: string };
  try {
    body = (await request.json()) as { password?: string };
  } catch {
    return Response.json({ error: "Invalid JSON" }, { status: 400 });
  }
  const pw = typeof body.password === "string" ? body.password : "";
  if (!pw) return Response.json({ error: "Missing password" }, { status: 400 });

  const result = await verifyPasswordAndSetSession(pw);
  if (!result.ok) {
    // Generic 401 on either "not configured" or "wrong password" — don't
    // leak which.
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }
  return Response.json({ ok: true });
}
