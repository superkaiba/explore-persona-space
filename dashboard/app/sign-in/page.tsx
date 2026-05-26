/**
 * Sign-in page — single shared site password.
 *
 * Submitting the correct `SITE_PASSWORD` sets the `eps_session` HttpOnly
 * cookie, which gates the protected routes (middleware in `proxy.ts`) and
 * unlocks editor-only server actions via `isEditorAuthed()`.
 */
import { getSitePassword } from "@/lib/auth";
import { PasswordForm } from "./PasswordForm";

export const dynamic = "force-dynamic";

export default async function SignIn({
  searchParams,
}: {
  searchParams: Promise<{ next?: string }>;
}) {
  const sp = await searchParams;
  const sitePwEnabled = Boolean(getSitePassword());
  const next = sp.next || "/";

  return (
    <article className="mx-auto max-w-md space-y-8">
      <header>
        <h1 className="text-xl font-semibold tracking-tight">Sign in</h1>
        <p className="mt-1 text-xs text-stone-500">
          Enter the site password to continue.
        </p>
      </header>

      {!sitePwEnabled ? (
        <p className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          Site password is not configured. Set <code>SITE_PASSWORD</code>{" "}
          (≥8 chars) in the dashboard&apos;s environment to enable.
        </p>
      ) : (
        <PasswordForm next={next} />
      )}
    </article>
  );
}
