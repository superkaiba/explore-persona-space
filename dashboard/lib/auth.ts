/**
 * Dashboard auth — two coexisting flows.
 *
 * 1. Editor cookie (existing): a shared `EDITOR_SECRET` gates the
 *    `/tasks/[id]/edit` route + every server action that mutates disk
 *    state. Single-user, defense-in-depth, was here before this file
 *    grew the magic-link bits below.
 *
 * 2. Magic-link session (new): for the `/updates` + sidecar/chat routes.
 *    `requestMagicLink` issues a 15-min JWT signed with `AUTH_SECRET`,
 *    emails it via Resend (or logs to console when `RESEND_API_KEY` is
 *    unset). `verifyMagicLinkAndSetSession` swaps it for a 30-day
 *    session JWT stored in the `eps_session` HttpOnly cookie.
 *    `requireSessionAuth` returns `{email}` on success or `null` on
 *    failure; callers turn `null` into a 401. When
 *    `DASHBOARD_AUTH_ENABLED !== "true"`, `requireSessionAuth` short-
 *    circuits to `{email: "dev@local"}` so local dev works without a
 *    Resend key. `middleware.ts` enforces the same gate at the edge.
 *
 * The two flows do NOT share cookies and are not interchangeable; the
 * `/sign-in` page exposes both as separate forms.
 */
import { cookies } from "next/headers";
import { SignJWT, jwtVerify, type JWTPayload } from "jose";

/* -------------------------------------------------------------------------- *
 * (1) Editor cookie — original behavior, unchanged.
 * -------------------------------------------------------------------------- */

export const EDITOR_COOKIE = "eps-editor";

export function getEditorSecret(): string | null {
  const s = process.env.EDITOR_SECRET;
  return typeof s === "string" && s.length >= 8 ? s : null;
}

export async function isEditorAuthed(): Promise<boolean> {
  const expected = getEditorSecret();
  if (!expected) return false;
  const jar = await cookies();
  return jar.get(EDITOR_COOKIE)?.value === expected;
}

export async function setEditorCookie(value: string): Promise<void> {
  const jar = await cookies();
  jar.set(EDITOR_COOKIE, value, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 24 * 30, // 30 days
  });
}

export async function clearEditorCookie(): Promise<void> {
  const jar = await cookies();
  jar.delete(EDITOR_COOKIE);
}

/* -------------------------------------------------------------------------- *
 * (2) Magic-link + session JWT.
 * -------------------------------------------------------------------------- */

export const SESSION_COOKIE = "eps_session";
const MAGIC_TTL_S = 15 * 60;
const SESSION_TTL_S = 30 * 24 * 60 * 60;

export type SessionUser = { email: string };

export function isAuthEnabled(): boolean {
  return process.env.DASHBOARD_AUTH_ENABLED === "true";
}

function authSecretBytes(): Uint8Array {
  const s = process.env.AUTH_SECRET;
  if (!s || s.length < 32) {
    throw new Error("AUTH_SECRET is missing or too short (need ≥32 chars)");
  }
  return new TextEncoder().encode(s);
}

function allowedEmails(): Set<string> {
  const raw = process.env.ALLOWED_EMAILS ?? "";
  return new Set(
    raw
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean),
  );
}

function siteUrl(): string {
  const raw = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3010";
  return raw.replace(/\/+$/, "");
}

type MagicPayload = JWTPayload & { sub: string; kind: "magic" };
type SessionPayload = JWTPayload & { sub: string; kind: "session" };

async function signMagic(email: string): Promise<string> {
  return await new SignJWT({ kind: "magic" } satisfies Pick<MagicPayload, "kind">)
    .setProtectedHeader({ alg: "HS256" })
    .setSubject(email)
    .setIssuedAt()
    .setExpirationTime(`${MAGIC_TTL_S}s`)
    .sign(authSecretBytes());
}

async function signSession(email: string): Promise<string> {
  return await new SignJWT({ kind: "session" } satisfies Pick<SessionPayload, "kind">)
    .setProtectedHeader({ alg: "HS256" })
    .setSubject(email)
    .setIssuedAt()
    .setExpirationTime(`${SESSION_TTL_S}s`)
    .sign(authSecretBytes());
}

/**
 * Issue a magic-link email. Returns `{ok: true, link}` (the caller can
 * choose to use that for tests; the API route discards it). When `email`
 * isn't in ALLOWED_EMAILS the function returns `{ok: false, reason}`
 * but the API route responds with the same shape regardless to avoid
 * leaking existence.
 */
export async function requestMagicLink(
  email: string,
): Promise<{ ok: true; link: string } | { ok: false; reason: string }> {
  const normalized = email.trim().toLowerCase();
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(normalized)) {
    return { ok: false, reason: "invalid_email" };
  }
  // Fail-closed: if ALLOWED_EMAILS is empty/missing, refuse all sign-ins
  // rather than letting anyone in. Critic P0-7.
  const allowed = allowedEmails();
  if (allowed.size === 0 || !allowed.has(normalized)) {
    return { ok: false, reason: "not_allowed" };
  }

  const token = await signMagic(normalized);
  const link = `${siteUrl()}/api/auth/verify?token=${encodeURIComponent(token)}`;

  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) {
    // Dev mode: print to server console; do NOT email.
    console.log(`[auth] magic link for ${normalized}: ${link}`);
    return { ok: true, link };
  }

  const from = process.env.MAGIC_LINK_FROM ?? "onboarding@resend.dev";
  // Import lazily so cold paths don't pay the cost.
  const { Resend } = await import("resend");
  const resend = new Resend(apiKey);
  const { error } = await resend.emails.send({
    from,
    to: normalized,
    subject: "Sign in to EPS dashboard",
    html: `<p>Click to sign in (valid for 15 minutes):</p><p><a href="${link}">${link}</a></p>`,
  });
  if (error) {
    console.error("[auth] resend send failed:", error);
    return { ok: false, reason: "send_failed" };
  }
  return { ok: true, link };
}

/**
 * Verify a magic token and mint a 30-day session cookie. Returns the
 * email on success.
 */
export async function verifyMagicLinkAndSetSession(
  token: string,
): Promise<{ ok: true; email: string } | { ok: false; reason: string }> {
  try {
    const { payload } = await jwtVerify(token, authSecretBytes());
    const p = payload as MagicPayload;
    if (p.kind !== "magic" || typeof p.sub !== "string") {
      return { ok: false, reason: "wrong_token_kind" };
    }
    const session = await signSession(p.sub);
    const jar = await cookies();
    jar.set(SESSION_COOKIE, session, {
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
      path: "/",
      maxAge: SESSION_TTL_S,
    });
    return { ok: true, email: p.sub };
  } catch (err) {
    return { ok: false, reason: err instanceof Error ? err.message : "verify_failed" };
  }
}

export async function clearSessionCookie(): Promise<void> {
  const jar = await cookies();
  jar.delete(SESSION_COOKIE);
}

/**
 * Read + verify the session cookie. Returns `null` when the cookie is
 * missing/invalid; callers MUST return 401 on null. The previous
 * "dev short-circuit" was the live exploit the critic ran (P0-1/2/3) —
 * anonymous users got a fake session and could spawn Claude subprocesses.
 * Auth is now mandatory regardless of `DASHBOARD_AUTH_ENABLED` (which
 * remains a flag for the `/sign-in` UI but never gates server enforcement).
 */
export async function requireSessionAuth(): Promise<SessionUser | null> {
  const jar = await cookies();
  const raw = jar.get(SESSION_COOKIE)?.value;
  if (!raw) return null;
  try {
    const { payload } = await jwtVerify(raw, authSecretBytes());
    const p = payload as SessionPayload;
    if (p.kind !== "session" || typeof p.sub !== "string") return null;
    return { email: p.sub };
  } catch {
    return null;
  }
}

/* -------------------------------------------------------------------------- *
 * (3) Shared-password sign-in (alternative to magic link).
 *
 * For users who haven't set up Resend. SITE_PASSWORD is the single shared
 * secret; matching it sets the SAME session cookie a magic link would, with
 * `sub` = "site-pw@local". Same TTL, same HttpOnly/Secure semantics.
 * -------------------------------------------------------------------------- */

export function getSitePassword(): string | null {
  const s = process.env.SITE_PASSWORD;
  return typeof s === "string" && s.length >= 8 ? s : null;
}

export async function verifyPasswordAndSetSession(
  password: string,
): Promise<{ ok: true; email: string } | { ok: false; reason: string }> {
  const expected = getSitePassword();
  if (!expected) return { ok: false, reason: "not_configured" };
  // Constant-time compare to avoid timing oracle.
  const a = new TextEncoder().encode(password);
  const b = new TextEncoder().encode(expected);
  if (a.length !== b.length) return { ok: false, reason: "wrong_password" };
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
  if (diff !== 0) return { ok: false, reason: "wrong_password" };

  const email = "site-pw@local";
  const session = await signSession(email);
  const jar = await cookies();
  jar.set(SESSION_COOKIE, session, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: SESSION_TTL_S,
  });
  return { ok: true, email };
}
