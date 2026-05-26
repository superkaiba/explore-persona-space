/**
 * Dashboard auth — single shared site password.
 *
 * Submitting the right `SITE_PASSWORD` to `POST /api/auth/password` mints
 * a 30-day session JWT signed with `AUTH_SECRET` and stores it in the
 * `eps_session` HttpOnly cookie. `requireSessionAuth()` reads + verifies
 * that cookie and returns `{email}` on success or `null` on failure;
 * callers turn `null` into a 401. `isEditorAuthed()` is the same check
 * with a friendlier name for the write-action gates.
 *
 * `middleware.ts` enforces the same cookie at the edge when
 * `DASHBOARD_AUTH_ENABLED === "true"`.
 */
import { cookies } from "next/headers";
import { SignJWT, jwtVerify, type JWTPayload } from "jose";

export const SESSION_COOKIE = "eps_session";
const SESSION_TTL_S = 30 * 24 * 60 * 60;

export type SessionUser = { email: string };

function authSecretBytes(): Uint8Array {
  const s = process.env.AUTH_SECRET;
  if (!s || s.length < 32) {
    throw new Error("AUTH_SECRET is missing or too short (need ≥32 chars)");
  }
  return new TextEncoder().encode(s);
}

type SessionPayload = JWTPayload & { sub: string; kind: "session" };

async function signSession(email: string): Promise<string> {
  return await new SignJWT({ kind: "session" } satisfies Pick<SessionPayload, "kind">)
    .setProtectedHeader({ alg: "HS256" })
    .setSubject(email)
    .setIssuedAt()
    .setExpirationTime(`${SESSION_TTL_S}s`)
    .sign(authSecretBytes());
}

export async function clearSessionCookie(): Promise<void> {
  const jar = await cookies();
  jar.delete(SESSION_COOKIE);
}

/**
 * Read + verify the session cookie. Returns `null` when the cookie is
 * missing/invalid; callers MUST return 401 on null.
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

/**
 * Write-action gate. Same check as `requireSessionAuth`, named for the
 * call sites that gate task edits, comment posting, and clean-result
 * mutations.
 */
export async function isEditorAuthed(): Promise<boolean> {
  return (await requireSessionAuth()) !== null;
}

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
