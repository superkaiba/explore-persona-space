/**
 * Cookie-based shared-secret auth for the dashboard editor.
 *
 * Single-user research tool: the only "user" is the researcher (Thomas).
 * Gates the /tasks/[id]/edit route + every server action that mutates
 * disk state. Cloudflare Access (zero-trust) can sit in front of this
 * later; the cookie check is a defense-in-depth.
 *
 * Setup:
 *   1. Set EDITOR_SECRET=<long random string> in the dashboard's env
 *      (systemd EnvironmentFile or `.env.local` for dev).
 *   2. Visit /sign-in?key=<the secret> once; the sign-in server action
 *      sets the `eps-editor` HttpOnly cookie.
 *   3. All subsequent edit/save requests carry the cookie.
 *
 * If EDITOR_SECRET is UNSET, edits are disabled (server actions return
 * { ok: false, error: "editor-disabled" }; the /edit route 404s). This
 * is the safe default for the public read-only path.
 */
import { cookies } from "next/headers";

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
