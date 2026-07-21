// 统一的 API 客户端, 自动带 token
const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export type Source = {
  doc_id?: string;
  chunk_id: string;
  content: string;
  score: number;
  metadata?: Record<string, unknown>;
};

export type ChatMode = "rag" | "agent";

export type SessionMeta = {
  session_id: string;
  title: string;
  kb_id: string;
  mode: string;
  created_at: number;
  updated_at: number;
  message_count: number;
};

export type HistoryMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  ts: number;
};

export type SessionDetail = SessionMeta & {
  messages: HistoryMessage[];
};

function authHeaders(token?: string): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ----- Auth -----
export async function register(username: string, email: string, password: string): Promise<{ access_token: string }> {
  return jsonOrThrow(await fetch(`${API}/api/v1/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, email, password }),
  }));
}

export async function login(username: string, password: string): Promise<{ access_token: string }> {
  return jsonOrThrow(await fetch(`${API}/api/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  }));
}

export async function devToken(): Promise<{ access_token: string }> {
  return jsonOrThrow(await fetch(`${API}/api/v1/auth/dev-token`, { method: "POST" }));
}

// ----- Chat (SSE 流式) -----
export async function* streamChat(params: {
  token?: string;
  question: string;
  session_id?: string;
  kb_id?: string;
  top_k?: number;
  mode?: ChatMode;
}): AsyncGenerator<{ type: string; data: unknown }> {
  const res = await fetch(`${API}/api/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(params.token) },
    body: JSON.stringify({
      question: params.question,
      session_id: params.session_id,
      kb_id: params.kb_id ?? "default",
      top_k: params.top_k ?? 5,
      mode: params.mode ?? "rag",
    }),
  });
  if (!res.body) throw new Error("SSE: no body");
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n\n");
    buf = lines.pop() || "";
    for (const chunk of lines) {
      const line = chunk.trim();
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      try {
        yield JSON.parse(payload);
      } catch {
        // ignore malformed line
      }
    }
  }
}

// ----- Documents -----
export async function uploadDocument(token: string, file: File, kb_id = "default") {
  const fd = new FormData();
  fd.append("file", file);
  return jsonOrThrow<{ doc_id: string; filename: string; chunks: number; status: string; message?: string }>(
    await fetch(`${API}/api/v1/documents/upload?kb_id=${encodeURIComponent(kb_id)}`, {
      method: "POST",
      headers: authHeaders(token),
      body: fd,
    }),
  );
}

export async function listDocuments(token: string, kb_id = "default") {
  return jsonOrThrow<{ kb_id: string; bm25_count: number; vector_count: number }>(
    await fetch(`${API}/api/v1/documents/list?kb_id=${encodeURIComponent(kb_id)}`, {
      headers: authHeaders(token),
    }),
  );
}

// ----- History -----
export async function listSessions(token: string): Promise<SessionMeta[]> {
  return jsonOrThrow(
    await fetch(`${API}/api/v1/history`, { headers: authHeaders(token) }),
  );
}

export async function getSession(token: string, session_id: string): Promise<SessionDetail> {
  return jsonOrThrow(
    await fetch(`${API}/api/v1/history/${session_id}`, { headers: authHeaders(token) }),
  );
}

export async function createSession(token: string, payload: { title?: string; kb_id?: string; mode?: string } = {}) {
  return jsonOrThrow<{ session_id: string }>(
    await fetch(`${API}/api/v1/history`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(token) },
      body: JSON.stringify(payload),
    }),
  );
}

export async function appendMessage(token: string, session_id: string, msg: HistoryMessage) {
  return jsonOrThrow<{ session_id: string; message_count: number }>(
    await fetch(`${API}/api/v1/history/${session_id}/append`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(token) },
      body: JSON.stringify(msg),
    }),
  );
}

export async function deleteSession(token: string, session_id: string) {
  return jsonOrThrow(
    await fetch(`${API}/api/v1/history/${session_id}`, {
      method: "DELETE",
      headers: authHeaders(token),
    }),
  );
}

export const API_BASE = API;

// ----- 客户端 token 存储 (与 AuthBar 共享) -----
const TOKEN_KEY = "unikb.token";
const USER_KEY = "unikb.user";

export function getStoredToken(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function getStoredUser(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(USER_KEY) || "";
}

export function setStoredToken(token: string, user: string): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, user);
  window.dispatchEvent(new Event("unikb:auth"));
}

export function clearStoredAuth(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  window.dispatchEvent(new Event("unikb:auth"));
}
