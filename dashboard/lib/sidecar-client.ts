type SidecarChatBody = {
  session_id?: string | null;
  provider?: string | null;
  messages: Array<{ role: string; content: string }>;
};

function normalizeBaseUrl(url: string) {
  return url.replace(/\/+$/, "");
}

function isSameOriginUrl(url: string) {
  if (typeof window === "undefined") return false;
  try {
    return new URL(url, window.location.href).origin === window.location.origin;
  } catch {
    return false;
  }
}

function chatUrls(sidecarUrl: string) {
  const direct = `${normalizeBaseUrl(sidecarUrl)}/chat`;
  const proxy = "/api/sidecar/chat";
  if (direct === proxy || isSameOriginUrl(direct)) return [direct];
  return [direct, proxy];
}

export async function postSidecarChat(
  sidecarUrl: string,
  token: string,
  body: SidecarChatBody,
) {
  const payload = JSON.stringify(body);
  let networkError: unknown = null;

  for (const url of chatUrls(sidecarUrl)) {
    try {
      return await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: payload,
      });
    } catch (error) {
      networkError = error;
    }
  }

  throw networkError ?? new Error("Unable to reach Claude Code sidecar");
}
