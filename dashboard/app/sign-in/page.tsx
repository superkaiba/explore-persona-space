import { getEditorSecret, isEditorAuthed } from "@/lib/auth";
import { redirect } from "next/navigation";
import { signIn } from "./actions";

export const dynamic = "force-dynamic";

export default async function SignIn({
  searchParams,
}: {
  searchParams: Promise<{ next?: string; error?: string; key?: string }>;
}) {
  const sp = await searchParams;
  const editorEnabled = Boolean(getEditorSecret());
  const next = sp.next || "/";

  // If already signed in, fast-path to `next`.
  if (editorEnabled && (await isEditorAuthed())) {
    redirect(next);
  }

  return (
    <article className="mx-auto max-w-md space-y-4">
      <h1 className="text-xl font-semibold tracking-tight">Editor sign-in</h1>
      {!editorEnabled && (
        <p className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          Editing is disabled. Set <code>EDITOR_SECRET</code> in the dashboard&apos;s
          environment to enable.
        </p>
      )}
      {editorEnabled && (
        <form action={signIn} className="space-y-3">
          <input type="hidden" name="next" value={next} />
          <label className="block">
            <span className="text-sm text-stone-700">Editor secret</span>
            <input
              type="password"
              name="key"
              required
              autoFocus
              defaultValue={sp.key || ""}
              className="mt-1 w-full rounded border border-stone-300 bg-white px-3 py-2 text-sm font-mono"
              placeholder="EDITOR_SECRET"
            />
          </label>
          {sp.error === "wrong" && (
            <p className="text-sm text-red-700">Wrong secret.</p>
          )}
          <button
            type="submit"
            className="w-full rounded bg-stone-900 px-3 py-2 text-sm font-medium text-white"
          >
            Sign in
          </button>
        </form>
      )}
      <p className="text-xs text-stone-500">
        Sets an HttpOnly cookie scoped to this dashboard. Cookie expires in 30 days.
      </p>
    </article>
  );
}
