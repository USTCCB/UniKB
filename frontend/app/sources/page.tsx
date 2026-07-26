"use client";
import { useEffect, useState } from "react";
import { API_BASE, getStoredToken } from "@/lib/api";

type DocChunk = {
  id: string;
  document: string;
  metadata?: Record<string, unknown>;
  distance?: number;
};

export default function SourcesPage() {
  const [token, setToken] = useState("");
  const [q, setQ] = useState("");
  const [results, setResults] = useState<DocChunk[]>([]);
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(10);
  const [error, setError] = useState("");

  useEffect(() => {
    setToken(getStoredToken());
  }, []);

  async function search() {
    if (!q.trim() || !token) return;
    setLoading(true);
    setError("");
    try {
      // 通过 RAG 流接口的 source 事件拉取, 更轻量: 直接走 chat/stream 但只关心 sources
      // 实际做法: 调用一个独立的检索接口 (下面 inline 了一个轻量 endpoint), 这里用 chat/stream 的 source 事件
      const res = await fetch(`${API_BASE}/api/v1/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ question: q, kb_id: "default", top_k: topK, mode: "rag" }),
      });
      if (!res.body) throw new Error("no body");
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      const out: DocChunk[] = [];
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n\n");
        buf = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          try {
            const payload = JSON.parse(line.slice(5).trim());
            if (payload.type === "source") {
              const d = payload.data;
              out.push({
                id: d.chunk_id || d.id,
                document: d.content,
                metadata: d.metadata || {},
                distance: 1 - (d.score || 0),
              });
            }
          } catch {
            /* ignore */
          }
        }
      }
      setResults(out);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  if (!token) {
    return (
      <div className="card">
        <div className="muted">请先登录。</div>
      </div>
    );
  }

  return (
    <div className="sources-page">
      <div className="card">
        <h3>知识库检索 (Hybrid Search + Rerank)</h3>
        <div className="row">
          <input
            type="text"
            placeholder="输入检索关键词..."
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && search()}
          />
          <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
            {[5, 10, 20].map((n) => (
              <option key={n} value={n}>top_k={n}</option>
            ))}
          </select>
          <button onClick={search} disabled={loading || !q.trim()}>
            {loading ? "检索中..." : "检索"}
          </button>
        </div>
        {error && <div className="err">{error}</div>}
        <div className="muted" style={{ marginTop: 8 }}>
          走 hybrid_search (BM25 + 向量 + RRF) → Cross-Encoder 重排, 只取 source 事件
        </div>
      </div>

      <div className="card">
        <h3>命中片段 ({results.length}) </h3>
        {results.length === 0 && <div className="muted">暂无结果</div>}
        {results.map((r, i) => (
          <div key={i} className="hit">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="mono">{r.id}</span>
              <span className="muted">distance={Number(r.distance ?? 0).toFixed(3)}</span>
            </div>
            <div className="text">{r.document}</div>
            <div className="meta">
              {Object.entries(r.metadata || {}).slice(0, 4).map(([k, v]) => (
                <span key={k} className="chip">{k}: {String(v)}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
