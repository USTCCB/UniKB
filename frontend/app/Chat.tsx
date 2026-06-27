"use client";
import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type Source = { chunk_id: string; content: string; score: number };

export default function Chat() {
  const [token, setToken] = useState("");
  const [q, setQ] = useState("UniKB 支持哪些 LLM？");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(false);

  async function ask() {
    if (!q.trim()) return;
    setLoading(true);
    setAnswer("");
    setSources([]);
    try {
      const res = await fetch(`${API}/api/v1/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        body: JSON.stringify({ question: q, kb_id: "default", top_k: 5, mode: "rag" }),
      });
      if (!res.body) return;
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          try {
            const payload = JSON.parse(line.slice(5).trim());
            if (payload.type === "token") {
              setAnswer((a) => a + (payload.data?.text || ""));
            } else if (payload.type === "source") {
              setSources((s) => [...s, payload.data]);
            } else if (payload.type === "error") {
              setAnswer((a) => a + `\n[error] ${payload.data?.msg}`);
            }
          } catch {}
        }
      }
    } catch (e: any) {
      setAnswer(`请求失败：${e?.message || e}`);
    } finally {
      setLoading(false);
    }
  }

  async function upload(file: File) {
    if (!token) { alert("请先填 token"); return; }
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${API}/api/v1/documents/upload?kb_id=default`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: fd,
    });
    const j = await r.json();
    alert(`入库成功：${j.filename}  chunks=${j.chunks}`);
  }

  return (
    <div>
      <div className="card">
        <div className="muted">JWT Token（dev 模式可调 /api/v1/auth/dev-token 拿）</div>
        <input type="text" placeholder="eyJ..." value={token} onChange={(e) => setToken(e.target.value)} />
      </div>
      <div className="card">
        <div className="muted">上传文档（PDF / Markdown / DOCX / TXT / 图片）</div>
        <input type="file" onChange={(e) => e.target.files && upload(e.target.files[0])} />
      </div>
      <div className="card">
        <textarea rows={3} value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={ask} disabled={loading}>{loading ? "生成中..." : "发送"}</button>
        </div>
      </div>
      {sources.length > 0 && (
        <div className="card">
          <div className="muted" style={{ marginBottom: 8 }}>引用来源（Top {sources.length}）</div>
          {sources.map((s, i) => (
            <div key={i} className="source">[{i + 1}] score={s.score?.toFixed?.(3) ?? s.score}  {s.content}</div>
          ))}
        </div>
      )}
      {answer && (
        <div className="card">
          <div className="muted" style={{ marginBottom: 8 }}>回答</div>
          <div className="answer">{answer}</div>
        </div>
      )}
    </div>
  );
}
