const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function getSessionId() {
  const key = "rag_session_id";
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(key, id);
  }
  return id;
}

function headers(extra = {}) {
  return { "X-Session-Id": getSessionId(), ...extra };
}

export async function health() {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}

export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/documents`, {
    method: "POST",
    headers: headers(),
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed (${res.status})`);
  }
  return res.json();
}

export async function deleteDocument(name) {
  const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(name)}`, {
    method: "DELETE",
    headers: headers(),
  });
  if (!res.ok) throw new Error(`Failed to remove "${name}"`);
  return res.json();
}

export async function documentStats() {
  const res = await fetch(`${API_BASE}/api/documents/stats`, { headers: headers() });
  return res.json();
}

/**
 * Streams the agent's response. The backend currently emits one `final`
 * event once the ReAct agent finishes (agent reasoning isn't token-level
 * streamable the way a raw completion is), but this is kept as an SSE
 * reader so a future incremental-token backend change is a drop-in swap.
 */
export async function chatWithAgent({ question, history, model, topK, maxTokens }, onFinal) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      question,
      history,
      model,
      top_k: topK,
      max_tokens: maxTokens,
    }),
  });

  if (!res.ok) {
    const contentType = res.headers.get("content-type") || "";
    const errPayload = contentType.includes("application/json") ? await res.json() : await res.text();
    throw new Error(errPayload.detail || errPayload.error || String(errPayload));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const event of events) {
      if (event.startsWith("event: error")) {
        const dataLine = event.split("\n").find((line) => line.startsWith("data: "));
        const payload = dataLine ? JSON.parse(dataLine.slice(6)) : { error: "Unknown agent error" };
        throw new Error(payload.error);
      }
      const dataLine = event.split("\n").find((line) => line.startsWith("data: "));
      if (!dataLine) continue;
      const data = dataLine.slice(6);
      if (data === "[DONE]") continue;

      const parsed = JSON.parse(data);
      if (parsed.type === "final") {
        onFinal(parsed.result);
      }
    }
  }
}
