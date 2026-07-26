"use client";
import { useEffect, useState } from "react";
import {
  Source,
  appendMessage,
  createSession,
  listSessions,
  streamChat,
  getStoredToken,
} from "@/lib/api";
import Link from "next/link";

type Msg = {
  role: "user" | "assistant";
  content: string;
  sources: Source[];
};

export default function ChatPage() {
  const [token, setToken] = useState("");
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<"rag" | "agent">("rag");
  const [topK, setTopK] = useState(5);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setToken(getStoredToken());
  }, []);

  async function ensureSession() {
    if (sessionId) return sessionId;
    if (!token) return null;
    const r = await createSession(token, { title: "新会话", mode });
    setSessionId(r.session_id);
    return r.session_id;
  }

  async function ask() {
    const text = q.trim();
    if (!text || !token) return;
    setQ("");

    const sid = await ensureSession();
    if (!sid) return;

    const userMsg: Msg = { role: "user", content: text, sources: [] };
    const asstMsg: Msg = { role: "assistant", content: "", sources: [] };
    setMessages((m) => [...m, userMsg, asstMsg]);
    setLoading(true);

    // 落库 user message
    try {
      await appendMessage(token, sid, {
        role: "user",
        content: text,
        ts: Date.now() / 1000,
      });
    } catch {
      /* ignore */
    }

    let collected = "";
    try {
      for await (const evt of streamChat({
        token,
        question: text,
        session_id: sid,
        mode,
        top_k: topK,
      })) {
        if (evt.type === "token") {
          const t = (evt.data as any)?.text ?? "";
          collected += t;
          setMessages((m) =>
            m.map((it, idx) =>
              idx === m.length - 1 ? { ...it, content: it.content + t } : it,
            ),
          );
        } else if (evt.type === "source") {
          const src = evt.data as Source;
          setMessages((m) =>
            m.map((it, idx) =>
              idx === m.length - 1 ? { ...it, sources: [...it.sources, src] } : it,
            ),
          );
        } else if (evt.type === "error") {
          collected += `\n[error] ${(evt.data as any)?.msg ?? ""}`;
          setMessages((m) =>
            m.map((it, idx) => (idx === m.length - 1 ? { ...it, content: collected } : it)),
          );
        }
      }
    } catch (e: any) {
      collected += `\n[error] ${e?.message ?? e}`;
      setMessages((m) =>
        m.map((it, idx) => (idx === m.length - 1 ? { ...it, content: collected } : it)),
      );
    } finally {
      setLoading(false);
      // 落库 assistant message
      try {
        await appendMessage(token, sid, {
          role: "assistant",
          content: collected,
          sources: asstMsg.sources,
          ts: Date.now() / 1000,
        });
      } catch {
        /* ignore */
      }
    }
  }

  function newChat() {
    setSessionId(null);
    setMessages([]);
  }

  if (!token) {
    return (
      <div className="card">
        <div className="muted">请先在右上角登录或获取 dev token。</div>
      </div>
    );
  }

  return (
    <div className="chat-page">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <div className="row">
          <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
            <option value="rag">RAG 模式</option>
            <option value="agent">多 Agent</option>
          </select>
          <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
            {[3, 5, 8, 10].map((n) => (
              <option key={n} value={n}>top_k={n}</option>
            ))}
          </select>
          <span className="muted">session: {sessionId ?? "(新会话)"}</span>
        </div>
        <button className="ghost" onClick={newChat}>新会话</button>
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div className="muted" style={{ padding: 24, textAlign: "center" }}>
            开始提问，比如「UniKB 支持哪些 LLM？」或「如何启用 LangFuse？」
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={"bubble " + m.role}>
            <div className="role">{m.role === "user" ? "你" : "助手"}</div>
            <div className="content">{m.content || (loading && i === messages.length - 1 ? "▍" : "")}</div>
            {m.sources.length > 0 && (
              <details className="sources">
                <summary>引用 {m.sources.length} 条</summary>
                {m.sources.map((s, j) => (
                  <div key={j} className="source">
                    <span className="score">{Number(s.score).toFixed(3)}</span>
                    <span className="text">{s.content}</span>
                  </div>
                ))}
              </details>
            )}
          </div>
        ))}
      </div>

      <div className="composer">
        <textarea
          rows={2}
          placeholder="输入问题..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
          }}
        />
        <button onClick={ask} disabled={loading || !q.trim()}>
          {loading ? "生成中..." : "发送"}
        </button>
      </div>

      <div className="muted" style={{ marginTop: 8, fontSize: 11 }}>
        提示: ⌘/Ctrl + Enter 发送。历史会话在 <Link href="/history">History</Link>。
      </div>
    </div>
  );
}
