"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export function PasswordForm({ next }: { next: string }) {
  const router = useRouter();
  const [pw, setPw] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const res = await fetch("/api/auth/password", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ password: pw }),
      });
      if (res.ok) {
        router.push(next);
        router.refresh();
        return;
      }
      const body = (await res.json().catch(() => ({}))) as { error?: string };
      setError(body.error ?? "Sign-in failed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-3">
      <label className="block">
        <span className="sr-only">Site password</span>
        <input
          type="password"
          value={pw}
          onChange={(e) => setPw(e.target.value)}
          required
          autoFocus
          className="w-full rounded border border-stone-300 bg-white px-3 py-2 text-sm font-mono"
          placeholder="Site password"
          disabled={busy}
        />
      </label>
      {error && <p className="text-sm text-red-700">{error}</p>}
      <button
        type="submit"
        disabled={busy}
        className="w-full rounded bg-stone-900 px-3 py-2 text-sm font-medium text-white disabled:opacity-60"
      >
        {busy ? "Signing in…" : "Sign in"}
      </button>
    </form>
  );
}
