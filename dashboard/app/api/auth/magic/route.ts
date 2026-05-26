/**
 * POST /api/auth/magic  body: {email}
 *
 * Responds with `{ok: true}` regardless of whether the email is in
 * ALLOWED_EMAILS — we don't want to leak which addresses are valid.
 * Real errors (bad payload, missing AUTH_SECRET) still return non-200.
 */
import { requestMagicLink } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ error: "invalid_json" }, { status: 400 });
  }

  const email = typeof (payload as { email?: unknown })?.email === "string"
    ? ((payload as { email: string }).email)
    : "";
  if (!email) {
    return Response.json({ error: "email_required" }, { status: 400 });
  }

  try {
    await requestMagicLink(email);
  } catch (err) {
    // AUTH_SECRET missing / Resend init failure — surface as 500, NOT
    // as "ok: true" because the operator needs to see it.
    const message = err instanceof Error ? err.message : "magic_link_failed";
    return Response.json({ error: message }, { status: 500 });
  }

  return Response.json({ ok: true });
}
