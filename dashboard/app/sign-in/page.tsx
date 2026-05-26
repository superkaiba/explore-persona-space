/**
 * Sign-in page — exposes both auth flows side by side.
 *
 * 1. Editor secret (existing): shared-key cookie that unlocks
 *    /tasks/[id]/edit + every server action that writes to disk. Used
 *    daily by the researcher.
 * 2. Magic-link session (new, optional): email-based JWT cookie that
 *    gates /updates and the sidecar/chat routes when
 *    DASHBOARD_AUTH_ENABLED=true. In dev (auth disabled), this form
 *    is shown but only useful for previewing the flow — the cookie
 *    isn't required by middleware.
 */
import { getEditorSecret, isAuthEnabled, isEditorAuthed } from "@/lib/auth";
import { redirect } from "next/navigation";
import { signIn } from "./actions";
import { MagicLinkForm } from "./MagicLinkForm";

export const dynamic = "force-dynamic";

const ERROR_MESSAGES: Record<string, string> = {
  missing_token: "Magic link is missing its token.",
  wrong_token_kind: "That link isn't a valid sign-in token.",
  invalid_email: "That email address doesn't look right.",
  not_allowed: "That email is not on the allow-list.",
  send_failed: "Couldn't email the link. Check the server logs.",
};

export default async function SignIn({
  searchParams,
}: {
  searchParams: Promise<{ next?: string; error?: string; key?: string }>;
}) {
  const sp = await searchParams;
  const editorEnabled = Boolean(getEditorSecret());
  const authEnabled = isAuthEnabled();
  const next = sp.next || "/";
  const error = sp.error ?? null;

  // Editor fast-path: if already authed, jump to `next`.
  if (editorEnabled && (await isEditorAuthed())) {
    redirect(next);
  }

  return (
    <article className="mx-auto max-w-md space-y-8">
      <header>
        <h1 className="text-xl font-semibold tracking-tight">Sign in</h1>
        <p className="mt-1 text-xs text-stone-500">
          Two coexisting flows. Editor secret unlocks task editing; the magic
          link gates the updates page and chat routes when the auth gate is on.
        </p>
      </header>

      {error && error !== "wrong" && (
        <p className="rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-900">
          {ERROR_MESSAGES[error] ?? `Sign-in error: ${error}`}
        </p>
      )}

      <section className="space-y-3">
        <h2 className="text-sm font-semibold tracking-tight text-stone-700">
          Editor secret
        </h2>
        {!editorEnabled && (
          <p className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
            Editing is disabled. Set <code>EDITOR_SECRET</code> in the
            dashboard&apos;s environment to enable.
          </p>
        )}
        {editorEnabled && (
          <form action={signIn} className="space-y-3">
            <input type="hidden" name="next" value={next} />
            <label className="block">
              <span className="sr-only">Editor secret</span>
              <input
                type="password"
                name="key"
                required
                defaultValue={sp.key || ""}
                className="w-full rounded border border-stone-300 bg-white px-3 py-2 text-sm font-mono"
                placeholder="EDITOR_SECRET"
              />
            </label>
            {error === "wrong" && (
              <p className="text-sm text-red-700">Wrong secret.</p>
            )}
            <button
              type="submit"
              className="w-full rounded bg-stone-900 px-3 py-2 text-sm font-medium text-white"
            >
              Sign in as editor
            </button>
          </form>
        )}
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold tracking-tight text-stone-700">
          Magic link
        </h2>
        {!authEnabled && (
          <p className="rounded border border-stone-300 bg-stone-50 px-3 py-2 text-xs text-stone-700">
            Auth gate is OFF (<code>DASHBOARD_AUTH_ENABLED=false</code>). The
            magic-link cookie isn&apos;t required for any route right now;
            this form is just to preview the flow.
          </p>
        )}
        <MagicLinkForm />
      </section>

      <p className="text-xs text-stone-500">
        Both flows set HttpOnly cookies scoped to this dashboard.
      </p>
    </article>
  );
}
