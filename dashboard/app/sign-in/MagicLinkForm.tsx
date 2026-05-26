"use client";

import { useState } from "react";

type State = { status: "idle" | "sending" | "sent" | "error"; message?: string };

export function MagicLinkForm() {
  const [email, setEmail] = useState("");
  const [state, setState] = useState<State>({ status: "idle" });

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!email.trim() || state.status === "sending") return;
    setState({ status: "sending" });
    try {
      const res = await fetch("/api/auth/magic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim() }),
      });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { error?: string };
        setState({ status: "error", message: body.error ?? `HTTP ${res.status}` });
        return;
      }
      setState({ status: "sent" });
    } catch (err) {
      setState({
        status: "error",
        message: err instanceof Error ? err.message : "network_error",
      });
    }
  }

  return (
    <form onSubmit={onSubmit} className="space-y-3">
      <label className="block">
        <span className="sr-only">Email address</span>
        <input
          type="email"
          required
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          placeholder="you@example.com"
          className="w-full rounded border border-stone-300 bg-white px-3 py-2 text-sm"
        />
      </label>
      <button
        type="submit"
        disabled={state.status === "sending"}
        className="w-full rounded bg-stone-900 px-3 py-2 text-sm font-medium text-white disabled:opacity-50"
      >
        {state.status === "sending" ? "Sending…" : "Email me a sign-in link"}
      </button>
      {state.status === "sent" && (
        <p className="rounded border border-emerald-300 bg-emerald-50 px-3 py-2 text-sm text-emerald-900">
          Check your email for a sign-in link. If your email isn&apos;t on the
          allow-list, no link will be sent.
        </p>
      )}
      {state.status === "error" && (
        <p className="rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-900">
          {state.message}
        </p>
      )}
    </form>
  );
}
