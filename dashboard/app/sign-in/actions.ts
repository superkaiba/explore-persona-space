"use server";

import { redirect } from "next/navigation";
import { getEditorSecret, setEditorCookie } from "@/lib/auth";

export async function signIn(formData: FormData): Promise<void> {
  const key = String(formData.get("key") || "");
  const next = String(formData.get("next") || "/");
  const expected = getEditorSecret();
  if (!expected) {
    redirect("/sign-in?error=disabled");
  }
  if (key !== expected) {
    redirect(`/sign-in?error=wrong&next=${encodeURIComponent(next)}`);
  }
  await setEditorCookie(key);
  redirect(next);
}
